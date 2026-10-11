# PIE-009I communications-latency / authority oracle evidence

Date: 2026-09-12

## Scope

This evidence note records the independent synthetic reference semantics in `scripts/pie-009i-comms-authority-oracle.py`.

The purpose is to prevent recovery campaigns from assuming instantaneous Earth control or allowing remote/advisory authority to bypass local essential-service protection.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

The oracle imports no Symthaea module.

## Reference semantics

- local safety/autonomy acts from current local state and continues during a communications outage;
- remote commands carry issue, nominal-arrival and expiry steps;
- a command cannot execute before arrival;
- a closed link delays delivery but does not grant future knowledge;
- an expired command fails closed;
- commands may carry an expected state-version precondition;
- local state changes can make a later-arriving command stale;
- advisory messages are recorded but are not authoritative;
- Earth/operator authority cannot override local essential-service protection;
- local failed-state guards can reject remote requests to re-enable nonessential work;
- duplicate IDs, invalid timing and unknown actions fail closed.

## Executed synthetic fixtures

The self-test demonstrates:

1. a remote command is not applied before its arrival step;
2. communications loss delays delivery until after expiry, causing rejection;
3. an intervening local failure changes state version and invalidates a stale command;
4. Earth/operator authority cannot disable protected essential-service behavior;
5. advisory input remains non-authoritative;
6. local safety behavior continues while the link is closed;
7. a remote nonessential-enable request is rejected while a local failure remains;
8. a local repair followed by a fresh version-matched remote command can succeed;
9. future link/failure differences do not change earlier local actions;
10. invalid timing and duplicate command IDs fail closed.

## Important limitations

This is not a spacecraft command protocol, networking simulator, mission-operations system, optimal-control solver, cryptographic authorization design, reliability model, or hardware-control implementation. Timing, failures and state transitions are synthetic references.

The current oracle does not model packet loss probabilities, clock uncertainty, multi-hop relay networks, Byzantine behavior, command authentication, operator conflicts, telemetry compression, or human factors. Those should remain separate layers.

## Promotion boundary

The intended path is:

`independent communications/authority oracle -> production authority interface -> cross-check with PIE-009H non-anticipative policies -> PIE-009E shock campaigns -> evidence-bounded Moon/Mars autonomy exercises`

Remote intelligence should remain advisory or scoped by explicit authority; local hard safety constraints remain the final guard for essential industrial services.
