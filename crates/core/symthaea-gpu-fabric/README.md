# symthaea-gpu-fabric

Backend-neutral semantic GPU execution contracts for Symthaea.

This crate intentionally starts with a deterministic CPU reference executor. It defines the semantic operation plan, resource limits, backend identity, kernel identity, and execution receipt that native Vulkan and browser WebGPU implementations will later share.

## First operation

The first contract is binary HDC binding:

`HdcBindXor` = bitwise XOR over canonical bit-packed hypervectors.

The CPU reference executor is the qualification oracle. A future accelerated backend must produce the same semantic result for the same plan and inputs, while proving that it actually executed on the claimed backend.

## Evidence boundary

Receipts bind the operation, plan digest, kernel identity/digest, input digest, output digest, backend, determinism mode, and acceleration claim.

The CPU reference backend is never reported as accelerated. Receipt verification rejects an inconsistent acceleration claim.

Resource validation occurs before execution and bounds input bytes, output bytes, dispatch invocations, vector dimensions, and canonical tail bits.

## Intended backend layering

```
semantic operation plan
        |
        +--> CPU reference
        +--> WebGPU / wgpu
        +--> native Vulkan
```

The semantic contract does not depend on any graphics API and does not introduce a parallel provenance ontology.

## Execution dependency graph

The `ExecutionGraph` models semantic resource hazards independently of a
backend. Nodes declare resource reads/writes and typed edges express
read-after-write, write-after-read, and write-after-write ordering.

The graph is validated fail-closed:

- cycles and unknown nodes are rejected;
- duplicate resource declarations and dependencies are rejected;
- dependency kinds must agree with the declared accesses;
- every read/write conflict must be ordered by the resulting DAG;
- graph identifiers are bounded;
- topological order and graph digest are independent of insertion order.

The graph digest is a semantic schedule identity. Vulkan synchronization2 and
timeline semaphores, WebGPU encoder ordering, and Prism compositor scheduling
should be lowerings of this graph rather than independent dependency models.

## Acceleration claims

Backend identity and acceleration are separate facts. A software Vulkan
qualification can legitimately produce a Vulkan execution receipt with
`accelerated=false`. An `accelerated=true` receipt requires concrete
implementation, device, and driver evidence.
