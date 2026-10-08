# Creditism as a Mycelix/Symthaea Mechanism Profile

**Status:** research / proposed qualification boundary  
**Scope:** Creditism as a candidate financial-system mechanism family for SYM-FIN-SYS-002 (#5606)  
**Related:** Symthaea #5606, #5607; Mycelix #3072, #3073, #3074

## Purpose

This document defines how the Common Planet Creditism architecture can be represented as a **falsifiable mechanism profile** inside the Mycelix/Symthaea financial-system scenario program.

It does **not** endorse Creditism, reject conventional finance, or treat Symthaea as an economic or political authority.

The central design rule is:

> Compare economic functions and accounting consequences, not ideological labels.

Creditism is unusually useful for this program because it changes several coupled primitives at once: who receives purchasing power, whether purchasing power is transferable, when it deletes, how contribution is recognized, how communities allocate shared capacity, and how productive assets are stewarded.

That coupling is also the primary modeling hazard. A simulation that labels the whole bundle "Creditism" and produces one outcome cannot tell us which mechanism caused it.

Therefore the implementation should decompose Creditism into independently switchable, explicitly named mechanisms before recombining them.

## 1. Current reference semantics

Common Planet currently describes Creditism as a system in which Personal Credit is issued to people, additional Personal Credit is associated with contribution, and Credit used at defined points of purchase deletes rather than becoming the seller's balance. The current architecture also separates Community Credit for shared purposes and Bonus Credit for group outcomes, and describes commons/stewardship arrangements for housing and productive assets.

Sources:

- https://common-planet.org/creditism
- https://common-planet.org/creditism/architecture
- https://common-planet.org/creditism/transition

Important current semantics for modeling:

- Personal Credit is non-transferable.
- Personal Credit is not described as interest-bearing, collateralizable, investable for autonomous return, or inheritable as concentrated power.
- Personal Credit may remain in a person's account for future use; it is not ordinary demurrage.
- Marketplace purchases delete the purchaser's Personal Credit.
- In the Exchange, buyer Credit deletes and the seller is separately credited up to their prior paid amount, preventing resale gain within the stated rule.
- Contribution recognition is a separate issuance process from consumer purchasing.
- Community Credit is a separate collective allocation stream.
- Bonus is a separate outcome-recognition stream.
- The Standard Pay Scale is publicly specified/revisable rather than negotiated by employers or buyers.
- Common Planet explicitly lists verification failure, allocation capture, issuance/calibration failure, and persistent misdirection of productive capacity as failure modes.

This means "Creditism" should not be represented as one scalar variable such as \`creditism=true\`.

## 2. The key accounting insight: two coupled circuits

The most important conceptual improvement is to model Creditism as two coupled but distinct circuits.

### Consumer-access circuit

~~~text
Personal Credit issuance
        |
        v
   household balance
        |
        v
  consumer purchase
        |
        v
     Credit deletion
~~~

### Contribution-recognition circuit

~~~text
verified contribution
        |
        v
contribution issuance
        |
        v
person / group balance
~~~

The buyer's spending event and the producer's recognition event are therefore not assumed to be the same accounting event.

This is materially different from the ordinary intuition:

~~~text
buyer payment -> seller receipt -> seller balance
~~~

The simulation must preserve the distinction.

A useful formal state variable is:

~~~text
PC_stock(t+1)
=
PC_stock(t)
+ existence_issuance
+ contribution_issuance
+ bonus_issuance
+ exchange_seller_issuance
- marketplace_deletion
- exchange_buyer_deletion
- other_declared_deletions
~~~

The exact fixture/profile must determine whether all terms are present.

For Exchange transactions, the stated rule can produce a matched deletion and seller re-credit, so the net outstanding Personal Credit can remain unchanged for that event.

The harder case is contribution issuance. There is no automatic accounting identity saying:

~~~text
consumer spending = producer reward
~~~

The model must therefore make the contribution rule explicit rather than silently treating producer recognition as sale revenue.

## 3. Why this matters for Mycelix accounting

Mycelix FIN-SYS-003 already requires stock-flow and counterpart integrity and rejects "new borrowing = income," asset-price gains being treated as productive output, and one-sided financial creation.

Creditism requires an analogous distinction between:

~~~text
access/deletion flow
!=
contribution-recognition flow
~~~

The simulator should never infer a producer's contribution issuance from the buyer's purchase merely because both events involve the same good or service.

Recommended event types:

~~~text
PersonalCreditIssuedForExistence
PersonalCreditIssuedForContribution
CommunityCreditIssued
BonusCreditIssued
MarketplaceCreditDeleted
ExchangeCreditDeleted
ExchangeSellerCreditIssued
CreditismRecognitionAdjustment
~~~

Each event must carry:

- issuer;
- recipient;
- mechanism profile;
- time;
- amount/unit;
- subject/product/project identity where applicable;
- evidence/verification reference;
- authority scope;
- accounting treatment;
- whether the event changes a stock, a flow, or both;
- uncertainty/indeterminate status.

## 4. Preserve the price-information problem

A major open research question is whether deleting consumer purchasing power destroys useful information that conventional monetary settlement currently carries.

Hayek's classic argument is specifically about dispersed knowledge: prices can communicate compact signals about changing scarcity and opportunity costs to decentralized actors. That argument does not prove conventional prices are optimal, but it establishes a real coordination function that an alternative architecture must address.

Source:

- https://www.aeaweb.org/aer/top20/35.4.519-530.pdf

Therefore the Creditism profile must separate at least four information channels:

~~~text
DemandSignal
ObservedPrice
ScarcitySignal
ContributionRecognition
~~~

Do not let:

~~~text
Credit deletion
~~~

automatically stand in for:

~~~text
market demand information
~~~

A strong demo should run paired worlds where:

- buyer demand is unchanged but producer-recognition rules differ;
- producer-recognition is unchanged but demand changes;
- prices are flexible;
- prices are administratively constrained;
- resource scarcity changes without demand changing;
- demand changes without physical scarcity changing.

The key output is not "which system wins." It is whether the same useful coordination information remains available, becomes noisier, or moves into another institution.

## 5. Creditism is not simply demurrage

This distinction should be explicit.

Demurrage schemes make holding the monetary unit itself costly over time. Current Creditism documentation instead allows Personal Credit to remain available for future use and makes the decisive constraint **non-transferability plus deletion at defined uses**.

That means a person may accumulate intertemporal purchasing capacity for their own future consumption without thereby obtaining a transferable financial claim over other people's future labor.

This creates a testable distinction:

~~~text
personal temporal saving
!=
transferable wealth claim
~~~

A useful adversarial fixture should therefore compare:

1. ordinary transferable savings;
2. non-transferable Personal Credit savings;
3. demurrage currency;
4. zero-store-of-value transaction tokens.

Measure separately:

- intertemporal smoothing;
- liquidity preference;
- concentration of claims;
- bargaining power;
- ability to finance long-lived assets;
- ability to transfer purchasing power across persons.

Recent literature on demurrage is useful as historical/comparative background, but it should not be treated as evidence about Creditism itself.

## 6. The deepest unresolved issue: capital formation

The hardest part of the Creditism model is not household purchasing power. It is long-horizon coordination.

Conventional financial systems encode future claims through debt, equity and related instruments. Creditism instead proposes community-directed capacity and commons/stewardship mechanisms.

The simulator therefore needs an explicit capital-formation layer:

~~~text
ProjectProposal
-> ResourceRequirement
-> FundingAllocation
-> Construction / R&D / Training
-> CapitalAssetCreation
-> OperatingCapacity
-> Maintenance / Replacement
~~~

The system must answer, experimentally rather than rhetorically:

- How is a project financed before it produces output?
- Who bears construction risk?
- Who absorbs cost overruns?
- Who decides whether a failed project should continue?
- How are maintenance obligations financed after completion?
- How are scarce productive assets allocated among competing projects?
- How are projects with long payback periods compared with projects with immediate benefits?
- What replaces transferable claims as the mechanism for intertemporal commitment?
- How does the system behave when several regions need the same machine, skilled team, or energy-intensive input simultaneously?

These questions are especially important because Mycelix #3073 already distinguishes financing from actual productive-capital formation.

## 7. Capital allocation must remain evidence-bound

Never map:

~~~text
Credit issued -> investment -> productivity
~~~

as an automatic causal chain.

The Mycelix bridge should preserve distinctions such as:

~~~text
ProductiveCapitalFormation
MaintenanceReplacementInvestment
WorkingCapital
ResearchAndDevelopment
HumanCapabilityInvestment
NewHousingOrConstruction
ExistingRealAssetAcquisition
ExistingFinancialClaimAcquisition
DebtRefinancing
DebtServiceOrInterest
HouseholdConsumption
PublicServiceOrInfrastructure
InventoryAccumulation
CrossBorderTransfer
MixedAllocation
UnknownAllocation
~~~

A Creditism scenario should be able to produce a large amount of Personal or Community Credit while still showing:

~~~text
productive capacity increase = unknown
~~~

That is a feature of epistemic integrity, not a failure of the model.

## 8. Commons governance should be modeled as a separate mechanism

The commons/stewardship component should not be smuggled into the monetary profile.

Create an independent governance parameterization:

~~~text
AssetTitleMode:
  Private
  Cooperative
  CommunityStewardship
  Public
  Hybrid

UseRightMode:
  Ownership
  Lease
  TimeBoundStewardship
  CooperativeAllocation

GovernanceMode:
  Market
  OnePersonOneVote
  Delegated
  Polycentric
  Mixed
~~~

Then recombine these dimensions with the monetary mechanism.

This is consistent with the empirical commons literature: durable common-pool institutions tend to depend on explicit boundaries, locally appropriate rules, collective choice, accountable monitoring, graduated sanctions, conflict resolution, recognized rights to organize, and nested coordination. Those principles do not identify one universal implementation, but they provide useful adversarial dimensions for the governance simulator.

Sources:

- https://www.nobelprize.org/prizes/economic-sciences/2009/illustrated-information/?print=1
- https://www.sciencedirect.com/science/article/pii/S0167268112002697

For Mycelix, this maps naturally onto evidence-bound governance, accountable authority, appeals, and nested institutions.

## 9. Verification is a first-class attack surface

The current Common Planet architecture explicitly leaves open:

- what qualifies as contribution;
- how completion is verified;
- how disputes are handled;
- how verification avoids becoming surveillance.

This should become one of the strongest synthetic attack suites.

Adversarial agents should attempt:

~~~text
measurement gaming
credential inflation
duplicate contribution
low-value high-frequency activity
valuable low-observability work
collusion
sybil activity
strategic task decomposition
outcome laundering
verification-panel capture
~~~

The simulator should compare:

~~~text
observed contribution
vs
latent contribution
~~~

and vary measurement error, observability, verifier incentives, and appeal effectiveness.

This is where Mycelix's epistemic fabric is potentially much more valuable than simply implementing a Credit wallet: the system can keep the evidence, authority, disagreement, appeal, and uncertainty structures separate from the economic rule itself.

## 10. Governance capture is the likely substitution effect

The strongest adversarial question is not merely:

> "Does Creditism prevent wealth concentration?"

It is:

> "If transferable financial accumulation is constrained, where does scarce economic power migrate?"

Candidate substitutes include:

~~~text
resource access
housing access
Community Credit allocation
project-selection authority
verification authority
pay-scale governance
reputation
information asymmetry
technical gatekeeping
infrastructure ownership/control
~~~

A successful simulation must track concentration across all of these.

Otherwise the system could pass a "wealth equality" metric while reproducing concentrated power through a different substrate.

Recommended outputs:

- purchasing-power concentration;
- scarce-resource access concentration;
- governance-decision concentration;
- verification authority concentration;
- infrastructure control concentration;
- information advantage concentration;
- appeal-resolution concentration.

No single "power concentration score."

## 11. Transition is not a conversion problem only

Common Planet's current transition architecture correctly identifies that an alternative currency cannot become a full replacement merely by changing the unit of account. External rent, utilities, fuel, taxes, contracts, and imported goods constrain transition.

The Symthaea transition model should therefore preserve:

~~~text
legacy claims
legacy contracts
foreign liabilities
physical imports
institutional migration costs
stranded assets
stranded financial claims
temporary dual-currency operation
legal recognition constraints
~~~

A transition fixture should be able to fail even when its steady state is attractive.

The accounting boundary is:

~~~text
opening balance sheet
-> explicit conversion / migration events
-> transition balance sheets
-> successor architecture
~~~

No silent liability deletion.

The UN 2025 SNA is appropriate as an external interoperability/reference framework because it is now the international statistical standard for national accounts and explicitly maintains integrated flow, stock, institutional-sector, financial-account and balance-sheet structures. It should remain a mapping reference, not a hidden authority over Mycelix semantics.

Source:

- https://unstats.un.org/unsd/nationalaccount/sna2025.asp

## 12. Interoperability architecture

The clean boundary is:

~~~text
Common Planet Creditism semantics
            |
            v
Creditism Mechanism Adapter
            |
            v
Mycelix Accounting Profile
            |
            +--> Evidence / Identity / Provenance
            |
            +--> Governance / Authority / Appeals
            |
            +--> Resource & Commons State
            |
            v
Symthaea Scenario Engine
            |
            v
Independent Oracle / Qualification Harness
~~~

No layer should claim authority belonging to another.

In particular:

- Common Planet semantics define the candidate mechanism.
- Mycelix defines interoperable identity/evidence/accounting/governance primitives.
- Symthaea explores consequences under explicit assumptions.
- The independent harness decides whether a synthetic fixture passed its declared theorem.
- None of these layers declares the political desirability of Creditism.

## 13. Minimal demonstration economy

For the first Common Planet-facing demo, use a small synthetic economy rather than a live economic system.

Suggested population:

~~~text
1,000 people
200 households
40 producers
10 community accounts
5 productive assets
5 common resources
3 imported inputs
1 external economy
~~~

Agent types:

~~~text
households
workers
producers
project councils
resource stewards
verifiers
governance participants
external suppliers
~~~

Run at least four profiles over the same initial physical world:

~~~text
A: conventional debt/market
B: mutual/risk-sharing
C: Creditism
D: hybrid
~~~

Creditism itself should be factored into toggles:

~~~text
non-transferable personal purchasing power
deletion at purchase
separate contribution issuance
community allocation stream
administered contribution rates
commons stewardship
outcome bonus
~~~

This lets us ask whether outcomes depend on one component or on the bundle.

## 14. Required scenario families

### Benign controls

- productive conventional borrowing;
- profitable SME expansion;
- healthy maturity transformation;
- successful equity-financed innovation;
- effective commons governance;
- high-value low-observability care work.

### Creditism-specific stress

- excessive Personal Credit issuance against fixed essentials;
- contribution measurement gaming;
- verifier capture;
- Community Credit allocation capture;
- simultaneous demand surge;
- long-horizon capital project;
- failed project / stranded asset;
- maintenance under low visible reward;
- imported energy shortage;
- external trade settlement problem.

### Information stress

- highly dispersed local knowledge;
- rapidly changing scarcity;
- missing observations;
- conflicting demand signals;
- stale prices;
- strategic misreporting;
- localized supply shock.

### Transition stress

- debt claims greater than replacement productive capacity;
- currency mismatch;
- foreign-denominated liabilities;
- contract non-conversion;
- temporary dual systems;
- migration bottleneck;
- legal/administrative lag.

## 15. Qualification invariants

Freeze these before simulation results are interpreted:

~~~text
borrowing != income
financial-claim creation != real-capital creation
asset-price gain != productive output
consumer deletion != producer payment
contribution issuance != sale revenue
community allocation != guaranteed productive success
scarcity signal != moral value
low inequality != absence of power concentration
low debt != good system
low prices != high welfare
high output != sustainability
simulation dominance != political recommendation
~~~

Add a particularly important equivalence test:

~~~text
Creditism world
+
mechanism-preserving renaming
->
same scenario result
~~~

and a decomposition test:

~~~text
same monetary rule
+
different governance rule
->
only declared governance-sensitive outputs change
~~~

This prevents implementation coupling from masquerading as economics.

## 16. Scientific claim ceiling

A successful first implementation can establish only things such as:

- the mechanism profile is internally specified;
- the synthetic accounting identities reconcile;
- the implementation behaves according to the declared Creditism rules;
- specified adversarial fixtures expose or fail to expose specified weaknesses;
- results differ or remain equivalent under controlled mechanism changes;
- the model preserves uncertainty and abstention under missing information.

It cannot establish:

- that Creditism works in a real economy;
- that Creditism is more humane, efficient, sustainable, or democratic in general;
- that capitalism or debt finance is intrinsically defective;
- that a simulation predicts actual transition outcomes;
- that Common Planet's proposed governance architecture is secure at production scale;
- that Symthaea can resolve political value conflicts objectively.

## 17. Recommended implementation order

The safest path is:

~~~text
1. Creditism semantic adapter
2. independent event/accounting profile
3. paired demand-vs-recognition fixtures
4. capital-formation fixtures
5. governance/verification attack corpus
6. transition fixtures
7. multi-agent synthetic economy
8. deterministic baseline
9. stochastic/adaptive agents
10. Symthaea comparative analysis
~~~

Do not begin with a learned policy.

The first result worth showing Common Planet is a transparent, replayable "mechanism microscope" that makes every issuance, deletion, contribution recognition, allocation decision, resource constraint, and governance decision inspectable.

## Bottom line

The strongest opportunity is not to build a Creditism wallet.

It is to build a **neutral experimental substrate in which Creditism can be run without being trusted**.

That gives Common Planet something valuable even if some of its proposed mechanisms fail:

~~~text
proposal
-> explicit mechanism
-> accounting
-> adversarial agents
-> controlled shocks
-> evidence-bound outputs
-> failure localization
~~~

And it gives Mycelix something broader than a single ideological economic implementation:

~~~text
different economic architectures
+
shared accounting
+
shared evidence semantics
+
shared governance/audit boundaries
+
independent experimental qualification
~~~

That is the architecture most likely to make the Common Planet/Symthaea collaboration scientifically useful rather than merely promotional.
