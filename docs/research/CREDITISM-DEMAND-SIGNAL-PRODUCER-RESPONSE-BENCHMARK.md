# Demand-Signal and Producer-Response Decoupling Benchmark

**Status:** research design v0  
**Related:** #7054, #7046, #7031, #7032, #7033, #7039, #7042, #7044

## Research question

> Can useful decentralized demand and scarcity information survive when transaction prices no longer transfer purchasing power to producers?

Current Creditism architecture explicitly separates buyer spending/deletion from contribution recognition and allows price to remain an access/scarcity signal. It also describes product selection through community purposes and quality competition rather than conventional producer revenue. https://common-planet.org/creditism/architecture

This makes an empirically interesting boundary between:

~~~text
price as access/scarcity signal
price as producer revenue
contribution recognition
capacity funding
product selection
~~~

These functions should not be collapsed.

## Literature anchor

Hayek's classic formulation describes prices as compressed information that lets producers respond to changes they do not directly observe. https://www.aeaweb.org/aer/top20/35.4.519-530.pdf

Recent work on quality competition shows that producer incentives to pursue quality depend on how the production regime connects quality, demand, and profitability; changing that connection can change innovation incentives. https://academic.oup.com/ser/article/22/4/1605/7657004

Therefore removal of seller revenue must be tested rather than treated as a mere accounting change.

## Paired mechanisms

### Conventional settlement

- buyer payment becomes seller purchasing power;
- price affects demand;
- seller revenue influences entry, expansion, exit, and investment;
- quality can be rewarded through higher revenue.

### Creditism profile

- buyer Personal Credit deletes;
- seller does not receive the buyer's spending balance;
- contribution recognition is a separate issuance event;
- Community Credit may fund shared inputs;
- price remains available as an access/scarcity observation;
- product purposes are selected through declared governance mechanisms.

Hold the physical production process constant wherever possible.

## Required state

Record separately:

- posted/listed price;
- purchaser Credit deleted;
- seller recognition;
- contribution Credit issued;
- Community Credit allocation;
- demand;
- observed scarcity;
- inventory;
- physical capacity;
- producer information state;
- producer belief state;
- production quantity;
- product quality;
- entry/exit;
- expansion/contraction;
- innovation;
- maintenance;
- replacement.

## Information ablations

### A — price present, direct demand absent

Producers observe prices but not the complete demand vector.

Measure whether production tracks the physical demand state.

### B — direct demand present, price absent

Producers observe demand quantities but not price.

Measure whether removing the price compression changes responsiveness or information burden.

### C — both present

Baseline information condition.

### D — both absent

Placebo/low-information control.

The causal comparison is:

~~~text
same physical world
+ different information channel
-> different producer response
~~~

## Revenue-decoupling treatment

Hold demand and physical capacity fixed.

Vary whether producer purchasing power is linked to buyer expenditure.

Measure:

- production response;
- entry;
- exit;
- expansion;
- contraction;
- capacity investment;
- quality innovation;
- product diversity;
- maintenance;
- response latency.

This isolates the effect of the revenue channel from the effect of transaction demand itself.

## Quality fixture

Construct identical producers with:

- low-quality/high-volume output;
- high-quality/lower-volume output;
- costly quality improvement;
- low observability;
- high observability.

Compare whether the institutional treatment changes the emergence and persistence of quality improvements.

Do not define quality as one scalar unless the profile explicitly does so; retain multidimensional quality where applicable.

## Capacity-investment fixture

Demand rises permanently, but new capacity requires a long lead time.

Measure how the system funds and selects capacity expansion when:

- revenue provides the funding signal;
- contribution recognition provides individual purchasing power;
- Community Credit funds shared capital;
- governance allocates project resources.

Any path from demand observation to physical expansion must be explicit.

## Information-loss attack

Remove one information channel at a time:

- price;
- direct demand;
- scarcity;
- inventory;
- quality evidence;
- contribution evidence.

An aggregate output change is not sufficient to identify which information function was lost. The event record must expose the channel and downstream response.

## Strategic manipulation

Allow selected agents to manipulate:

- listed price;
- reported demand;
- quality score;
- contribution verification;
- shortage reports.

Compare reported state against physical state.

Required distinction:

~~~text
reported improvement
!=
physical improvement
~~~

This is particularly important because performance metrics can create incentives to optimize the measured indicator rather than the underlying objective.

## Positive controls

Include:

- conventional price/revenue producer control;
- contribution-pay control;
- Community Credit capital-allocation control;
- mixed revenue/contribution control.

No information architecture is treated as inherently superior.

## Qualification boundary

A PASS establishes only the declared information and producer-response behavior under the frozen synthetic profile.

It does not establish:

- that markets possess uniquely valuable information;
- that Creditism preserves all price-information functions;
- that non-price coordination is superior;
- that producer incentives are adequate in real economies;
- that any production architecture is socially desirable.