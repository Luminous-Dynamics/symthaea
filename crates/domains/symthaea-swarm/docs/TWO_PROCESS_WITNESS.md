# Native two-process witness

This witness is the acceptance test for the hardened gossip layer. It must run as
two separate OS processes; an in-process pair is insufficient evidence for
endpoint, router, address-import, restart, and shutdown behavior.

## Host process

1. Load or create a persistent Iroh `SecretKey` outside this crate.
2. Create a `MemoryLookup` and install it on the endpoint builder.
3. Create an authenticated `TelepathicSocket` on a fixed topic.
4. Register `gossip_protocol()` on an Iroh `Router`.
5. Start `socket.run()`.
6. Call `invite_online(DEFAULT_ONLINE_WAIT)` and serialize the signed invitation.
7. Wait for `socket.wait_for_neighbors(1, timeout)`.
8. Broadcast one best-effort state message and one durable law proposal using
   `broadcast_tracked`.
9. Persist the returned message IDs in the witness log.
10. Wait for the joiner's signed response.
11. Shut down the socket and router cleanly.

## Join process

1. Read and verify the host invitation.
2. Load or create a separate persistent Iroh `SecretKey`.
3. Create a `MemoryLookup`, install it on the endpoint, and pass the same lookup to
   `from_invite_authenticated`.
4. Register the gossip protocol on a router and start the socket.
5. Wait for one neighbor.
6. Verify the host messages expose the host's signed `EndpointId` as `author`.
7. Broadcast a response carrying the joiner's application UUID.
8. Shut down cleanly.

## Required assertions

- both processes observe at least one neighbor;
- the invitation signature covers the topic and bootstrap addresses;
- an expired invitation is rejected;
- invitation addresses were imported into `MemoryLookup`;
- tampered envelopes are rejected with `InvalidEnvelope`;
- replaying the exact envelope increments duplicate metrics but does not redeliver;
- changing timestamp/message ID while reusing a session sequence is rejected;
- a restarted process using the same endpoint key and a new session ID may restart
  sequence numbering at one;
- a second endpoint claiming an already-bound UUID is rejected;
- `PinnedOnly` rejects an unenrolled but correctly signed endpoint;
- broadcasting before `run()` returns `NotRunning`;
- a best-effort queue overflow increments `best_effort_dropped`;
- a durable queue overflow ends the socket with `DurableQueueFull` without pausing
  the receive loop;
- a rate-limited neighbor produces a structured `RateLimited` rejection and metric;
- `BroadcastReceipt` is explicitly treated as local acceptance, not remote ack;
- shutdown reaches `SocketState::Stopped` without hanging.

## Restart witness

After the initial exchange:

1. persist the host and joiner Iroh keys and identity books;
2. stop both processes;
3. restart with the same keys and new socket sessions;
4. load the identity books in `PinnedOnly` mode;
5. rejoin with a new signed invitation;
6. confirm sequence `1` is accepted in the new session;
7. replay a captured envelope from the old session and confirm it is rejected or
   deduplicated while retained replay windows remain active.

Long-term cross-restart replay resistance requires persisting an application
message ledger, not only the in-memory transport window.

## Network fault extension

Repeat with relay-only address filtering, temporary network loss, process restart,
reordered/duplicated captures, a slow local consumer, malformed payload bursts,
and a rendezvous channel that attempts to tamper with an invitation.
