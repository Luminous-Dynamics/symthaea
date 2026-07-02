# Mycelix ERP: 18-Month Development Gantt Chart

**Version**: 1.0
**Date**: December 30, 2025
**Project Start**: January 1, 2026
**Target Completion**: June 30, 2027 (18 months)
**Team**: Sacred Trinity (Human + Claude Code + Local LLM)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Timeline Overview](#2-timeline-overview)
3. [Phase 0: Foundation](#phase-0-foundation-weeks-1-2)
4. [Phase 1: Financial Management](#phase-1-financial-management-weeks-3-14)
5. [Phase 2: Customer Relationship Management](#phase-2-customer-relationship-management-weeks-11-22)
6. [Phase 3: Manufacturing Resource Planning](#phase-3-manufacturing-resource-planning-weeks-19-30)
7. [Phase 4: Procurement](#phase-4-procurement-weeks-27-36)
8. [Phase 5: Human Resources](#phase-5-human-resources-weeks-33-44)
9. [Phase 6: Project Management](#phase-6-project-management-weeks-41-50)
10. [Phase 7: Asset Management](#phase-7-asset-management-weeks-47-56)
11. [Phase 8: Polish & Launch](#phase-8-polish--launch-weeks-53-60)
12. [Continuous Activities](#continuous-activities)
13. [Resource Allocation](#resource-allocation)
14. [Risk Management Timeline](#risk-management-timeline)
15. [Milestones & Deliverables](#milestones--deliverables)

---

## 1. Executive Summary

### Project Goals

Transform Mycelix-SupplyChain into **Mycelix ERP** - a complete, decentralized Enterprise Resource Planning system with 7 modules:

1. ✅ **SCM** - Supply Chain Management (v0.4.0 - Complete)
2. **FIN** - Financial Management (Phase 1)
3. **CRM** - Customer Relationship Management (Phase 2)
4. **MRP** - Manufacturing Resource Planning (Phase 3)
5. **HR** - Human Resources (Phase 5)
6. **PM** - Project Management (Phase 6)
7. **ASSET** - Asset Management (Phase 7)

### Timeline Summary

| Metric | Value |
|--------|-------|
| **Total Duration** | 60 weeks (15 months active development + 3 months buffer) |
| **Phases** | 8 (Foundation + 7 modules) |
| **Major Milestones** | 24 |
| **Total Tasks** | 180+ |
| **Parallel Work Streams** | Up to 3 concurrent |
| **Target v2.0 Release** | June 30, 2027 |

### Success Criteria

- ✅ All 7 modules production-ready
- ✅ >90% test coverage maintained
- ✅ <100ms API response time (p95)
- ✅ Comprehensive documentation
- ✅ 5+ pilot customers using full ERP
- ✅ $50K+ MRR (Monthly Recurring Revenue)

---

## 2. Timeline Overview

### Visual Timeline (Weeks 1-60)

```
Weeks:    |1 |5 |10|15|20|25|30|35|40|45|50|55|60|
          |==|==|==|==|==|==|==|==|==|==|==|==|==|
Phase 0:  [██]
Phase 1:  [  ][████████████]
Phase 2:       [    ][████████████]
Phase 3:              [    ][████████████]
Phase 4:                     [    ][█████████]
Phase 5:                          [    ][████████████]
Phase 6:                                   [    ][██████████]
Phase 7:                                        [    ][█████████]
Phase 8:                                                    [██████████]

Marketing:        [════════════════════════════════════════════════════]
Testing:               [════════════════════════════════════════════════]
Docs:                  [════════════════════════════════════════════════]

Legend:
██ = Active development
[  ] = Planning/Research
════ = Continuous activity
```

### Parallelization Strategy

```
Month 1-3:   FIN (primary focus)
Month 2-5:   FIN (finish) + CRM (start in month 3)
Month 4-7:   CRM (finish) + MRP (start in month 5)
Month 6-9:   MRP (finish) + Procurement (start in month 7)
Month 8-11:  Procurement (finish) + HR (start in month 9)
Month 10-13: HR (finish) + PM (start in month 11)
Month 12-15: PM (finish) + Asset (start in month 13)
Month 14-18: Asset (finish) + Polish & Launch
```

**Key Insight**: Modules have 2-month overlap to maximize parallelization while maintaining quality.

---

## Phase 0: Foundation (Weeks 1-2)

**Goal**: Restructure repository and build shared infrastructure

**Duration**: 2 weeks
**Team**: 1.0 FTE (Tristan + AI)

### Week 1: Repository Restructuring

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Rename repo to `mycelix-erp` | 0.5 | Tristan | None | Updated GitHub repo |
| Create Cargo workspace | 0.5 | Claude | None | `Cargo.toml` workspace |
| Move SCM to `services/scm/` | 1 | Claude | Workspace | Restructured directories |
| Create shared infrastructure skeleton | 1 | Claude | None | `shared/{crypto,auth,models,storage}/` |
| Update README.md with ERP vision | 1 | Tristan | None | New README |
| Create 18-month roadmap doc | 1 | Tristan | None | `ROADMAP.md` |

### Week 2: Shared Infrastructure (Auth + Models)

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Build `shared/auth` crate | 2 | Claude | None | JWT auth, RBAC |
| Build `shared/models` crate | 1 | Claude | None | Common types (Address, Money) |
| Build `shared/storage` crate | 1 | Claude | None | DB abstraction |
| Write tests for shared crates | 1 | Claude | All above | >90% coverage |

**Milestone**: 🎯 **Foundation Complete** (End of Week 2)
- ✅ Repository restructured
- ✅ Shared infrastructure ready
- ✅ All tests passing

---

## Phase 1: Financial Management (Weeks 3-14)

**Goal**: Build complete financial module (GL, AR, AP, Reporting)

**Duration**: 12 weeks
**Team**: 1.0 FTE
**Dependencies**: Phase 0 complete

### Weeks 3-4: Planning & Design

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Research financial workflows | 2 | Tristan | None | Requirements doc |
| Design database schema | 2 | Tristan + Claude | Requirements | Schema DDL |
| Design API endpoints | 2 | Claude | Schema | OpenAPI spec |
| Create domain models (Rust) | 2 | Claude | Schema | `fin/src/models/` |
| Write integration test plan | 2 | Tristan | Models | Test plan doc |

### Weeks 5-7: Core Implementation (GL + Chart of Accounts)

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement GL entry model | 2 | Claude | Domain models | `gl_entry.rs` |
| Implement Account model | 2 | Claude | Domain models | `account.rs` |
| Create database migrations | 2 | Claude | Schema | SQL migration files |
| Build GL API handlers | 3 | Claude | Models + DB | `/v1/fin/gl/*` endpoints |
| Write GL unit tests | 2 | Claude | Handlers | Test files |
| Write GL integration tests | 2 | Claude | Unit tests | E2E tests |
| Build TypeScript SDK (GL) | 2 | Claude | API | `sdk/fin/gl.ts` |

### Weeks 8-10: Accounts Receivable (Invoices)

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Invoice model | 2 | Claude | GL complete | `invoice.rs` |
| Implement InvoiceLineItem model | 1 | Claude | Invoice | `invoice_line_item.rs` |
| Create invoice migrations | 1 | Claude | Schema | SQL migrations |
| Build invoice API handlers | 4 | Claude | Models + DB | `/v1/fin/ar/*` endpoints |
| Implement VC generation for invoices | 2 | Claude | Crypto lib | Verifiable invoices |
| Write invoice tests | 3 | Claude | Handlers | Unit + integration tests |
| Build TypeScript SDK (AR) | 2 | Claude | API | `sdk/fin/ar.ts` |

### Weeks 11-12: Accounts Payable (Bills) + Payments

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Bill model | 2 | Claude | GL complete | `bill.rs` |
| Implement Payment model | 1 | Claude | Invoice + Bill | `payment.rs` |
| Create bill/payment migrations | 1 | Claude | Schema | SQL migrations |
| Build bill API handlers | 3 | Claude | Models + DB | `/v1/fin/ap/*` endpoints |
| Build payment API handlers | 2 | Claude | Models + DB | `/v1/fin/payments/*` |
| Write bill/payment tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (AP) | 1 | Claude | API | `sdk/fin/ap.ts` |

### Weeks 13-14: Reporting + Integration

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Balance Sheet report | 2 | Claude | GL | `/v1/fin/reports/balance-sheet` |
| Implement Income Statement report | 2 | Claude | GL | `/v1/fin/reports/income-statement` |
| Implement Cash Flow report | 2 | Claude | GL | `/v1/fin/reports/cash-flow` |
| Integrate FIN ↔ SCM (auto-create bills) | 2 | Claude | Both modules | Event handler |
| Write comprehensive FIN docs | 2 | Tristan | All above | User guide + API docs |
| Perform end-to-end testing | 2 | Tristan | All above | Test report |
| Deploy FIN beta to staging | 1 | Tristan | Tests pass | Staging environment |
| Beta test with 1-2 pilot customers | 7 | Tristan | Staging | Feedback report |

**Milestone**: 🎯 **FIN Module Complete** (End of Week 14)
- ✅ General Ledger functional
- ✅ Accounts Receivable (invoices) working
- ✅ Accounts Payable (bills) working
- ✅ Payment tracking implemented
- ✅ Financial reports generated
- ✅ Integration with SCM
- ✅ TypeScript SDK published
- ✅ Documentation complete
- ✅ 2+ pilot customers using FIN

---

## Phase 2: Customer Relationship Management (Weeks 11-22)

**Goal**: Build CRM module (Customers, Leads, Opportunities, Quotes)

**Duration**: 12 weeks
**Team**: 1.0 FTE
**Dependencies**: FIN module started (Week 11 overlap)
**Parallelization**: Overlaps with FIN weeks 11-14

### Weeks 11-12: Planning & Design

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Research CRM workflows | 2 | Tristan | None | Requirements doc |
| Design database schema | 2 | Claude | Requirements | Schema DDL |
| Design API endpoints | 2 | Claude | Schema | OpenAPI spec |
| Create domain models | 2 | Claude | Schema | `crm/src/models/` |
| Write integration test plan | 2 | Tristan | Models | Test plan |

### Weeks 13-15: Customer & Contact Management

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Customer model | 2 | Claude | Models | `customer.rs` |
| Implement Contact model | 1 | Claude | Customer | `contact.rs` |
| Create customer migrations | 1 | Claude | Schema | SQL migrations |
| Build customer API handlers | 3 | Claude | Models + DB | `/v1/crm/customers/*` |
| Write customer tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (Customers) | 2 | Claude | API | `sdk/crm/customers.ts` |
| Integrate customer creation → FIN | 2 | Claude | FIN module | Event handler |

### Weeks 16-18: Lead & Opportunity Management

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Lead model | 2 | Claude | Models | `lead.rs` |
| Implement Opportunity model | 2 | Claude | Models | `opportunity.rs` |
| Create lead/opportunity migrations | 1 | Claude | Schema | SQL migrations |
| Build lead API handlers | 3 | Claude | Models + DB | `/v1/crm/leads/*` |
| Build opportunity API handlers | 3 | Claude | Models + DB | `/v1/crm/opportunities/*` |
| Implement lead conversion flow | 2 | Claude | Leads + Customers | Convert endpoint |
| Write lead/opportunity tests | 2 | Claude | Handlers | Unit + integration |

### Weeks 19-21: Quote Management

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Quote model | 2 | Claude | Models | `quote.rs` |
| Implement QuoteLineItem model | 1 | Claude | Quote | `quote_line_item.rs` |
| Create quote migrations | 1 | Claude | Schema | SQL migrations |
| Build quote API handlers | 3 | Claude | Models + DB | `/v1/crm/quotes/*` |
| Implement VC generation for quotes | 2 | Claude | Crypto lib | Verifiable quotes |
| Write quote tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (Quotes) | 2 | Claude | API | `sdk/crm/quotes.ts` |
| Integrate quote → invoice (FIN) | 2 | Claude | FIN module | Event handler |

### Week 22: Documentation & Testing

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Write CRM documentation | 2 | Tristan | All above | User guide + API docs |
| Perform end-to-end testing | 2 | Tristan | All above | Test report |
| Deploy CRM beta to staging | 1 | Tristan | Tests pass | Staging environment |

**Milestone**: 🎯 **CRM Module Complete** (End of Week 22)
- ✅ Customer management functional
- ✅ Lead management working
- ✅ Opportunity tracking implemented
- ✅ Quote generation with VCs
- ✅ Integration with FIN (quotes → invoices)
- ✅ TypeScript SDK published
- ✅ Documentation complete

---

## Phase 3: Manufacturing Resource Planning (Weeks 19-30)

**Goal**: Build MRP module (BOMs, Work Orders, Inventory)

**Duration**: 12 weeks
**Team**: 1.0 FTE
**Dependencies**: CRM started (Week 19 overlap)
**Parallelization**: Overlaps with CRM weeks 19-22

### Weeks 19-20: Planning & Design

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Research manufacturing workflows | 2 | Tristan | None | Requirements doc |
| Design BOM structure | 2 | Tristan + Claude | Requirements | BOM schema |
| Design database schema | 2 | Claude | Requirements | Schema DDL |
| Design API endpoints | 2 | Claude | Schema | OpenAPI spec |
| Create domain models | 2 | Claude | Schema | `mrp/src/models/` |

### Weeks 21-23: Bill of Materials (BOM)

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement BOM model | 2 | Claude | Models | `bom.rs` |
| Implement Component model | 1 | Claude | BOM | `component.rs` |
| Implement ManufacturingStep model | 1 | Claude | BOM | `manufacturing_step.rs` |
| Create BOM migrations | 1 | Claude | Schema | SQL migrations |
| Build BOM API handlers | 3 | Claude | Models + DB | `/v1/mrp/boms/*` |
| Implement BOM versioning | 2 | Claude | BOM | Version control |
| Write BOM tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (BOM) | 2 | Claude | API | `sdk/mrp/bom.ts` |

### Weeks 24-26: Work Orders

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement WorkOrder model | 2 | Claude | BOM | `work_order.rs` |
| Create work order migrations | 1 | Claude | Schema | SQL migrations |
| Build work order API handlers | 4 | Claude | Models + DB | `/v1/mrp/work-orders/*` |
| Implement work order scheduling | 2 | Claude | WorkOrder | Scheduling logic |
| Write work order tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (WorkOrder) | 2 | Claude | API | `sdk/mrp/work-orders.ts` |
| Integrate work order → batch (SCM) | 2 | Claude | SCM module | Event handler |

### Weeks 27-29: Inventory Management

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Inventory model | 2 | Claude | Models | `inventory.rs` |
| Implement InventoryTransaction model | 1 | Claude | Inventory | `inventory_transaction.rs` |
| Create inventory migrations | 1 | Claude | Schema | SQL migrations |
| Build inventory API handlers | 3 | Claude | Models + DB | `/v1/mrp/inventory/*` |
| Implement stock level tracking | 2 | Claude | Inventory | Real-time updates |
| Write inventory tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (Inventory) | 2 | Claude | API | `sdk/mrp/inventory.ts` |
| Integrate inventory ↔ SCM | 2 | Claude | SCM module | Event handlers |

### Week 30: Documentation & Testing

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Write MRP documentation | 2 | Tristan | All above | User guide + API docs |
| Perform end-to-end testing | 2 | Tristan | All above | Test report |
| Deploy MRP beta to staging | 1 | Tristan | Tests pass | Staging |

**Milestone**: 🎯 **MRP Module Complete** (End of Week 30)
- ✅ BOM management functional
- ✅ Work order creation & tracking
- ✅ Inventory management
- ✅ Integration with SCM
- ✅ TypeScript SDK published
- ✅ Documentation complete

---

## Phase 4: Procurement (Weeks 27-36)

**Goal**: Build Procurement module (Purchase Orders, Vendor Management)

**Duration**: 10 weeks
**Team**: 0.5 FTE (simpler module)
**Dependencies**: MRP started (Week 27 overlap)
**Parallelization**: Overlaps with MRP weeks 27-30

### Weeks 27-28: Planning & Implementation

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Design procurement schema | 2 | Claude | None | Schema DDL |
| Implement Vendor model | 1 | Claude | Models | `vendor.rs` |
| Implement PurchaseOrder model | 2 | Claude | Models | `purchase_order.rs` |
| Create procurement migrations | 1 | Claude | Schema | SQL migrations |
| Build vendor API handlers | 2 | Claude | Models + DB | `/v1/procurement/vendors/*` |
| Build PO API handlers | 3 | Claude | Models + DB | `/v1/procurement/pos/*` |

### Weeks 29-35: Integration & Testing

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement PO approval workflow | 2 | Claude | PO | Multi-level approval |
| Write procurement tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK | 2 | Claude | API | `sdk/procurement/*.ts` |
| Integrate PO → Bill (FIN) | 2 | Claude | FIN module | Event handler |
| Integrate PO ↔ Inventory (MRP) | 2 | Claude | MRP module | Event handlers |
| Write documentation | 2 | Tristan | All above | Docs |
| Deploy to staging | 1 | Tristan | Tests | Staging |

### Week 36: Buffer

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Address feedback | 5 | Claude | Staging feedback | Improvements |

**Milestone**: 🎯 **Procurement Module Complete** (End of Week 36)
- ✅ Vendor management
- ✅ Purchase order creation & approval
- ✅ Integration with FIN (PO → Bill)
- ✅ Integration with MRP (inventory)

---

## Phase 5: Human Resources (Weeks 33-44)

**Goal**: Build HR module (Employees, Payroll, Time Tracking)

**Duration**: 12 weeks
**Team**: 1.0 FTE (compliance-heavy)
**Dependencies**: Procurement started (Week 33 overlap)
**Parallelization**: Overlaps with Procurement weeks 33-36

### Weeks 33-34: Planning & Design

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Research payroll regulations | 3 | Tristan | None | Compliance doc |
| Design employee schema | 2 | Claude | Requirements | Schema DDL |
| Design API endpoints | 2 | Claude | Schema | OpenAPI spec |
| Create domain models | 3 | Claude | Schema | `hr/src/models/` |

### Weeks 35-37: Employee Management

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Employee model | 2 | Claude | Models | `employee.rs` |
| Implement Department model | 1 | Claude | Models | `department.rs` |
| Create employee migrations | 1 | Claude | Schema | SQL migrations |
| Build employee API handlers | 4 | Claude | Models + DB | `/v1/hr/employees/*` |
| Implement onboarding workflow | 2 | Claude | Employee | Onboarding |
| Write employee tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (Employee) | 2 | Claude | API | `sdk/hr/employees.ts` |

### Weeks 38-40: Payroll

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement Payroll model | 3 | Claude | Employee | `payroll.rs` |
| Implement PayrollItem model | 1 | Claude | Payroll | `payroll_item.rs` |
| Create payroll migrations | 1 | Claude | Schema | SQL migrations |
| Build payroll API handlers | 4 | Claude | Models + DB | `/v1/hr/payroll/*` |
| Implement tax calculation | 3 | Claude | Payroll | Tax logic (US) |
| Write payroll tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK (Payroll) | 2 | Claude | API | `sdk/hr/payroll.ts` |

### Weeks 41-43: Time Tracking & Benefits

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Implement TimeEntry model | 2 | Claude | Employee | `time_entry.rs` |
| Implement Benefit model | 2 | Claude | Employee | `benefit.rs` |
| Create time/benefit migrations | 1 | Claude | Schema | SQL migrations |
| Build time tracking API | 3 | Claude | Models + DB | `/v1/hr/time/*` |
| Build benefits API | 3 | Claude | Models + DB | `/v1/hr/benefits/*` |
| Write time/benefit tests | 2 | Claude | Handlers | Unit + integration |
| Build TypeScript SDK | 2 | Claude | API | `sdk/hr/*.ts` |

### Week 44: Documentation & Testing

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Write HR documentation | 2 | Tristan | All above | Docs |
| Perform compliance review | 2 | External | HR module | Compliance report |
| End-to-end testing | 1 | Tristan | All above | Test report |

**Milestone**: 🎯 **HR Module Complete** (End of Week 44)
- ✅ Employee management
- ✅ Payroll processing
- ✅ Time tracking
- ✅ Benefits management
- ✅ Compliance verified

---

## Phase 6: Project Management (Weeks 41-50)

**Goal**: Build PM module (Projects, Tasks, Time Tracking)

**Duration**: 10 weeks
**Team**: 0.75 FTE (lighter module)
**Dependencies**: HR started (Week 41 overlap)
**Parallelization**: Overlaps with HR weeks 41-44

### Weeks 41-42: Planning & Core Models

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Design project schema | 2 | Claude | None | Schema DDL |
| Implement Project model | 2 | Claude | Schema | `project.rs` |
| Implement Task model | 2 | Claude | Project | `task.rs` |
| Create project migrations | 1 | Claude | Schema | SQL migrations |
| Build project API handlers | 3 | Claude | Models + DB | `/v1/pm/projects/*` |

### Weeks 43-48: Task Management & Time Tracking

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Build task API handlers | 3 | Claude | Task model | `/v1/pm/tasks/*` |
| Implement task dependencies | 2 | Claude | Tasks | Dependency graph |
| Implement project time tracking | 3 | Claude | HR time tracking | Time allocation |
| Build project reporting | 3 | Claude | Projects + Tasks | Dashboards |
| Write PM tests | 3 | Claude | All handlers | Unit + integration |
| Build TypeScript SDK | 2 | Claude | API | `sdk/pm/*.ts` |
| Integrate PM ↔ HR (time sync) | 2 | Claude | HR module | Event handler |
| Write documentation | 2 | Tristan | All above | Docs |

### Weeks 49-50: Testing & Polish

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| End-to-end testing | 2 | Tristan | All above | Test report |
| Performance optimization | 2 | Claude | Testing | Optimizations |
| Deploy to staging | 1 | Tristan | Tests | Staging |

**Milestone**: 🎯 **PM Module Complete** (End of Week 50)
- ✅ Project management
- ✅ Task tracking
- ✅ Time allocation
- ✅ Project reporting

---

## Phase 7: Asset Management (Weeks 47-56)

**Goal**: Build Asset module (Equipment, Maintenance)

**Duration**: 10 weeks
**Team**: 0.75 FTE
**Dependencies**: PM started (Week 47 overlap)
**Parallelization**: Overlaps with PM weeks 47-50

### Weeks 47-48: Planning & Core Models

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Design asset schema | 2 | Claude | None | Schema DDL |
| Implement Asset model | 2 | Claude | Schema | `asset.rs` |
| Implement MaintenanceRecord model | 2 | Claude | Asset | `maintenance_record.rs` |
| Create asset migrations | 1 | Claude | Schema | SQL migrations |
| Build asset API handlers | 3 | Claude | Models + DB | `/v1/asset/assets/*` |

### Weeks 49-54: Maintenance & Depreciation

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Build maintenance API | 3 | Claude | MaintenanceRecord | `/v1/asset/maintenance/*` |
| Implement depreciation calculation | 3 | Claude | Asset | Depreciation logic |
| Implement maintenance scheduling | 2 | Claude | Maintenance | Scheduler |
| Write asset tests | 3 | Claude | All handlers | Unit + integration |
| Build TypeScript SDK | 2 | Claude | API | `sdk/asset/*.ts` |
| Integrate assets → FIN (depreciation) | 2 | Claude | FIN module | Event handler |
| Write documentation | 2 | Tristan | All above | Docs |

### Weeks 55-56: Testing & Polish

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| End-to-end testing | 2 | Tristan | All above | Test report |
| Performance optimization | 2 | Claude | Testing | Optimizations |
| Deploy to staging | 1 | Tristan | Tests | Staging |

**Milestone**: 🎯 **Asset Module Complete** (End of Week 56)
- ✅ Asset tracking
- ✅ Maintenance scheduling
- ✅ Depreciation calculation
- ✅ Integration with FIN

---

## Phase 8: Polish & Launch (Weeks 53-60)

**Goal**: Production readiness, performance tuning, launch preparation

**Duration**: 8 weeks
**Team**: 1.5 FTE (ramp up for launch)
**Dependencies**: Most modules complete by Week 53

### Weeks 53-55: Performance & Security

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Load testing (k6) | 3 | Claude | All modules | Load test report |
| Performance optimization | 4 | Claude | Load tests | <100ms p95 response |
| Security audit (internal) | 3 | Tristan | All modules | Security report |
| Fix critical vulnerabilities | 3 | Claude | Audit | Patches |
| Penetration testing (external) | 5 | External | Security fixes | Pentest report |

### Weeks 56-58: Documentation & Marketing

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Update all module docs | 5 | Tristan | All modules | Complete docs |
| Create video tutorials | 5 | Tristan | Docs | 10+ tutorial videos |
| Write blog posts (launch series) | 3 | Tristan | Launch prep | 5 blog posts |
| Create case studies | 3 | Tristan | Pilot customers | 3 case studies |
| Prepare pitch deck | 2 | Tristan | Case studies | Investor deck |
| Design marketing website | 5 | External | Branding | mycelix.net/erp |

### Weeks 59-60: Launch!

| Task | Days | Assignee | Dependencies | Deliverable |
|------|------|----------|--------------|-------------|
| Final QA testing | 3 | Tristan + Claude | All above | QA report |
| Deploy to production | 1 | Tristan | QA pass | Production deployment |
| Launch announcement | 1 | Tristan | Deployment | Press release |
| HackerNews/Reddit posts | 1 | Tristan | Launch | Community engagement |
| Onboard 5+ new pilot customers | 5 | Tristan | Launch | 5+ customers |
| Monitor & hotfix issues | 5 | Claude | Launch | Stability |

**Milestone**: 🎯 **v2.0 Production Launch** (End of Week 60)
- ✅ All 7 modules production-ready
- ✅ Performance targets met (<100ms p95)
- ✅ Security audit passed
- ✅ Documentation complete
- ✅ 10+ pilot customers
- ✅ Public launch executed

---

## Continuous Activities

### Marketing & Sales (Weeks 3-60)

| Activity | Frequency | Assignee | Deliverable |
|----------|-----------|----------|-------------|
| Blog posts | 2/month | Tristan | 30+ posts |
| Social media (Twitter/LinkedIn) | 3/week | Tristan | Ongoing engagement |
| Customer interviews | 1/week | Tristan | Feedback reports |
| Sales calls | 2/week | Tristan | Pipeline growth |
| Newsletter | 1/month | Tristan | Email subscribers |

### Testing & QA (Weeks 5-60)

| Activity | Frequency | Assignee | Deliverable |
|----------|-----------|----------|-------------|
| Unit test writing | Daily | Claude | >90% coverage maintained |
| Integration test runs | Daily | CI/CD | Automated |
| Manual QA testing | Weekly | Tristan | QA reports |
| Performance regression tests | Weekly | Claude | Benchmark reports |
| Security scanning | Weekly | CI/CD | Vulnerability reports |

### Documentation (Weeks 5-60)

| Activity | Frequency | Assignee | Deliverable |
|----------|-----------|----------|-------------|
| API documentation updates | Per module | Claude | OpenAPI specs |
| User guide updates | Per module | Tristan | User docs |
| Code comments | Daily | Claude | Inline docs |
| Architecture decision records | As needed | Tristan | ADR docs |
| Changelog updates | Weekly | Tristan | CHANGELOG.md |

---

## Resource Allocation

### Team Composition

**Sacred Trinity Development Model**:
- **Human (Tristan)**: 1.0 FTE (vision, testing, documentation, customer engagement)
- **AI (Claude Code)**: Unlimited (code generation, architecture, testing)
- **Local LLM (Mistral-7B)**: Unlimited (domain expertise, code review)

### Infrastructure Budget

| Period | Monthly Cost | Annual Cost | Notes |
|--------|-------------|-------------|-------|
| **Months 1-6** | $200 | $1,200 | Development only |
| **Months 7-12** | $400 | $2,400 | Staging + testing |
| **Months 13-18** | $800 | $4,800 | Production + replicas |
| **Total Year 1-2** | - | **$8,400** | Infrastructure |

**Breakdown**:
- Database (PostgreSQL): $0 (self-hosted) → $200/mo (managed)
- Hosting (VPS): $100/mo → $400/mo
- CI/CD (GitHub Actions): $0 (free tier)
- Monitoring (Prometheus + Grafana): $0 (self-hosted)
- Domain & SSL: $50/year

### External Services Budget

| Service | Cost | When | Purpose |
|---------|------|------|---------|
| **Security Audit** | $5,000 | Week 55 | Penetration testing |
| **Legal (Compliance)** | $2,000 | Week 44 | HR/Payroll compliance |
| **Design (Marketing)** | $3,000 | Week 57 | Website & branding |
| **Total** | **$10,000** | One-time | External services |

---

## Risk Management Timeline

### High-Risk Periods

| Weeks | Risk | Probability | Impact | Mitigation |
|-------|------|-------------|--------|------------|
| **11-14** | FIN module complexity | Medium | High | Add 2 week buffer, prioritize testing |
| **19-22** | CRM-FIN integration issues | Medium | Medium | Early integration testing |
| **27-30** | MRP manufacturing logic | High | High | External domain expert review |
| **41-44** | HR compliance issues | Medium | High | Legal review, external audit |
| **55-58** | Security vulnerabilities | Low | Critical | External pentest, bug bounty |

### Risk Mitigation Strategies

**Technical Risks**:
- Maintain >90% test coverage throughout
- Daily CI/CD runs catch regressions early
- Weekly code reviews with AI
- Architectural Decision Records (ADRs) for major choices

**Schedule Risks**:
- 2-week buffer in each phase
- Parallel development where possible
- Scope reduction option (defer nice-to-haves)
- Can delay v2.0 launch by 1-2 months if needed

**Resource Risks**:
- Sacred Trinity model reduces single-person dependency
- Claude Code provides unlimited development capacity
- Budget buffer of $5,000 for unexpected costs

**Market Risks**:
- Early customer engagement (starting Week 5)
- Iterative feedback loops
- Pivot-ready architecture (modular design)

---

## Milestones & Deliverables

### Major Milestones

| Week | Milestone | Deliverables | Success Criteria |
|------|-----------|--------------|------------------|
| **2** | Foundation Complete | Workspace, shared libs | All tests passing |
| **14** | FIN Module v1.0 | Full financial module | 2+ pilot customers |
| **22** | CRM Module v1.0 | Complete CRM | Quote → Invoice flow works |
| **30** | MRP Module v1.0 | Manufacturing | BOM → Work Order → Batch |
| **36** | Procurement v1.0 | PO management | PO → Bill integration |
| **44** | HR Module v1.0 | Payroll & time tracking | Compliance verified |
| **50** | PM Module v1.0 | Project management | Time allocation working |
| **56** | Asset Module v1.0 | Asset tracking | Depreciation → FIN |
| **60** | **v2.0 Production Launch** | Full ERP suite | 10+ customers, <100ms p95 |

### Revenue Milestones

| Month | MRR Target | Customers | Notes |
|-------|------------|-----------|-------|
| **3** | $2,500 | 5 (FIN beta) | Early adopters |
| **6** | $7,500 | 15 (FIN + CRM) | Product-market fit |
| **9** | $15,000 | 30 (FIN + CRM + MRP) | Manufacturing customers |
| **12** | $25,000 | 50 (4 modules) | Expanded market |
| **15** | $40,000 | 80 (6 modules) | Enterprise interest |
| **18** | **$60,000** | **120 (Full ERP)** | **v2.0 Launch Success** |

**Annual Revenue Projection (Month 18)**: $60K MRR × 12 = **$720K ARR**

---

## Appendix: Task Dependencies Graph

```
Phase 0 (Foundation)
└── Phase 1 (FIN)
    ├── Phase 2 (CRM) [depends on FIN invoice integration]
    │   └── Phase 3 (MRP) [depends on CRM sales orders]
    │       ├── Phase 4 (Procurement) [depends on MRP inventory]
    │       └── Phase 5 (HR) [independent, can parallelize]
    │           └── Phase 6 (PM) [depends on HR time tracking]
    │               └── Phase 7 (Asset) [depends on FIN depreciation]
    └── Phase 8 (Polish & Launch) [depends on all modules]

Marketing (continuous, starts Week 3)
Testing (continuous, starts Week 5)
Documentation (continuous, starts Week 5)
```

---

## Appendix: Weekly Capacity Planning

**Assumptions**:
- 1 FTE = 40 hours/week = 5 days/week
- Sacred Trinity multiplier: 3x (AI assistance)
- Effective capacity: 1 FTE = 15 development days/week equivalent

**Capacity Allocation** (example for Week 15):

| Activity | FTE | Days/Week | Notes |
|----------|-----|-----------|-------|
| FIN final testing | 0.3 | 1.5 | Wrapping up Phase 1 |
| CRM customer management | 0.5 | 2.5 | Active Phase 2 development |
| Documentation | 0.1 | 0.5 | Continuous |
| Customer calls | 0.1 | 0.5 | Sales & feedback |
| **Total** | **1.0** | **5.0** | Full utilization |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-30 | Tristan Stoltz, Claude Code | Initial 18-month Gantt chart |

---

**End of 18-Month Gantt Chart**

**Next Steps**:
1. Review and approve timeline
2. Set up project management tool (Linear, Jira, or GitHub Projects)
3. Import milestones and tasks
4. Begin Phase 0 (Week 1)

**Questions? Concerns?** Discuss adjustments before starting Phase 0.

---

🎯 **Ready to build the world's best decentralized ERP!** 🚀
