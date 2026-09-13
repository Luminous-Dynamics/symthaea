# HLS episodic exact-gradient training

This note defines the first recurrent-learning protocol for the diagonal Holographic Liquid State (HLS) architecture.

## Objective

The goal is not to prove that HLS learning works by construction. The goal is to create a protocol in which any measured gain can be attributed cleanly to recurrent parameter learning rather than decoder capacity, data leakage, or ambiguous gradient semantics.

## Episode rule

For one generated world, HLS parameters remain fixed from the first observable initialization event through the final query.

During the episode:

1. recurrent state is reset;
2. `HlsEligibilityTrace` is reset;
3. every initialization/mutation event advances state and exact trace together;
4. every scored query uses the fixed parameter-free associative readout;
5. `dL/dh` is contracted with the exact trace to obtain `dL/dtheta`;
6. query gradients are accumulated without changing recurrent parameters.

Only after the final query is the episode-mean gradient applied.

## Optimizer boundary

The first optimizer is intentionally minimal:

- average query gradients within one world;
- global L2 clip;
- one gradient-descent step;
- explicit per-scalar absolute parameter bound.

If `G` is the mean episode gradient, the update is

`G_clipped = G * min(1, clip / ||G||)`

`theta_next = clamp(theta - eta G_clipped, -B, +B)`.

No Adam moments, learned schedules, replay buffers, or decoder weights are introduced in this tranche.

## Exactness conditions

The episode gradient is exact for the implemented diagonal-HLS recurrence under the exact eligibility-trace conditions:

- recurrent parameters are fixed throughout the episode;
- the global cross-coordinate norm limiter does not activate;
- contextual HLS is excluded from this exact local trace;
- events and query losses are deterministic functions of the fixed benchmark/codebook.

Experiments intended to use exact traces should set `state_norm_limit = infinity` and use a bounded odd activation such as tanh.

## Benchmark observability

Each world now begins with a shuffled observable initialization prefix that discloses every randomized entity location and object owner before scoring begins. Held-out worlds therefore differ in initial state, but no scored answer depends on a fact that was never presented to the model.

## Evaluation rule

Held-out evaluation clones and resets the trained/frozen cell and never mutates supplied parameters.

The primary comparison is therefore:

- frozen diagonal HLS on held-out worlds;
- exactly episode-trained diagonal HLS on the same held-out worlds.

The protocol deliberately does not encode an assertion that training must improve accuracy or loss. A negative or null result is valid evidence.

## Next qualification stage

Once CI qualifies the mechanics, run a multi-seed study with preregistered hyperparameters and report:

- held-out associative cosine loss;
- overall/current/historical/compositional accuracy;
- gradient norms and clipping frequency;
- parameter movement;
- binding-equivariance error after training;
- state dimension, trainable scalar count, and wall-clock/event throughput.

Only after this within-HLS learning ablation should stronger external baselines be added.
