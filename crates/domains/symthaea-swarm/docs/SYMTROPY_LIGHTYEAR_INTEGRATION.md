# Symtropy / Lightyear integration contract

`symtropy::SymtropyDirectAdapter` is the project-facing boundary between one
Lightyear connection entity and `DirectTransport`.

It intentionally does not depend on Bevy or Lightyear. A plugin should:

1. create one adapter per authenticated remote endpoint;
2. convert Lightyear output into a `SymtropyPacketClass`;
3. supply stable operation UUIDs for reliable control, checkpoints, robotics,
   and asset transfer;
4. call `flush_one` from an asynchronous networking task rather than blocking a
   Bevy schedule;
5. route `DirectEvent` values to the adapter bound to `message.author`;
6. drain `pop_inbound` into the corresponding Lightyear Link receive buffer;
7. treat reliable adapter overflow as a fatal health/readiness fault;
8. allow datagram overflow to be counted and dropped.

## Lane contract

| Packet class | Direct lane | Primitive | Operation UUID |
|---|---|---|---|
| Session control | control | reliable | required |
| Player input | player input | datagram | forbidden |
| State delta | state snapshot | datagram | forbidden |
| Checkpoint | state snapshot | reliable | required |
| Telemetry | telemetry | datagram | forbidden |
| Robotics command | robotics | reliable | required |
| Asset transfer | asset transfer | reliable | required |

Do not broadcast one Lightyear connection's packets to every peer. Each adapter
is endpoint-bound and rejects events from any other authenticated endpoint.

The current `symtropy-lightyear` archive uses a synchronous in-memory
`IrohTransport` placeholder and ignores send errors. It should be replaced by a
Tokio-owned `DirectTransport` plus these endpoint-bound adapters. The Bevy ECS
component should hold bounded Link-side queues and a command handle, not an Iroh
endpoint or async runtime.


## Reliable adapter overflow

A reliable direct ACK is issued when the transport event enters the direct
application channel, before the optional Symtropy adapter queue. If that second
queue is full, `InboundReliableQueueFull` returns ownership of the authenticated
`SymtropyPacket`. The caller must retain/retry that packet, apply it immediately,
or fail the connection. Discarding the returned packet would violate the
apply-once contract because later network retries receive a duplicate ACK.
