# Subterranean Team Operations Protocol

## Trust boundary

`symthaea-subterranean` accepts team messages only through typed ingestion APIs. The crate validates ranges, capacity, epoch, sequence, freshness, and state transitions. The caller must authenticate the sender and bind `AgentId` to a cryptographic identity before ingestion.

A transport should reject unauthenticated traffic before calling:

- `ingest_team_heartbeat`
- `merge_shared_tunnel_observation`
- `ingest_tunnel_reservation`
- `merge_mesh_link`
- `receive_rescue_request`
- rescue offer/acceptance transition methods

## Ordering

Heartbeats, map observations, reservations, and mesh links carry an epoch and sequence number. A new epoch supersedes every sequence from an earlier epoch. Within one epoch, sequence numbers must strictly increase.

Equal-version identical payloads are replays. Equal-version differing map or mesh payloads are equivocation and are rejected rather than resolved by arrival order.

## Freshness

Peer records and mesh links become non-authoritative after a bounded number of local steps. Stale information remains distinguishable from a weak current link; it does not create reachability or active distress authority.

## Map fusion

The shared map retains provenance per peer. Aggregation is intentionally conservative. Peer route evidence can worsen a return assessment but cannot make it more optimistic than local measurements.

## Tunnel reservations

The reference tunnel model is a depth interval. Opposing or holding reservations that overlap the local look-ahead corridor create a conflict. Priority order is:

1. emergency,
2. rescue,
3. return,
4. routine.

Equal priority is resolved by the lower `AgentId`. Runtime safety still arrests local motion on an imminent conflict even when the peer is expected to yield; right-of-way is not treated as collision immunity.

## Rescue consent

A distress heartbeat announces state only. Rescue mission authority requires:

1. a valid, current rescue request,
2. a feasible offer preserving the rescuer's return reserve,
3. acceptance by the requester,
4. an explicit begin transition.

Local physical hazards always override rescue intent.

## Evidence

Command-level evidence records the team state that affected each actuation decision. This allows reviewers to distinguish peer distress, link partition, occupancy conflict, accepted rescue, and actual post-arbitration commands.
